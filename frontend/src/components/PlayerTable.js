import React, { useState, useMemo } from 'react';
import { 
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender
} from '@tanstack/react-table';
import { useDraft } from '../contexts/DraftContext';
import { DRAFT_STATUS } from '../services/draftService';

const PlayerTable = ({ players, availableStats }) => {
  const { 
    draftPlayerForMe, 
    draftPlayerForOthers, 
    getPlayerDraftStatus 
  } = useDraft();

  console.error("rerendering table")
  
  // State for selected player
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  
  // State for visible columns
  const [visibleColumns, setVisibleColumns] = useState([
    'PLAYER_NAME', 'POSITIONS', 'RANK', 'HR', 'TB', 'RBI', 'SB', 'R'
  ]);
  
  // State for column selector visibility
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  
  // State for pagination
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: 25,
  });
  
  // Define columns
  const columns = useMemo(() => {
    const baseColumns = [
      {
        accessorKey: 'PLAYER_NAME',
        header: 'Name',
        cell: info => {
          const player = info.row.original;
          const status = getPlayerDraftStatus(player.PLAYER_ID);

          let statusClass = '';
          if (status === DRAFT_STATUS.DRAFTED_BY_YOU) {
            statusClass = 'drafted-by-you';
          } else if (status === DRAFT_STATUS.DRAFTED_BY_OTHERS) {
            statusClass = 'drafted-by-others';
          }

          return (
            <div className={`player-name ${statusClass}`}>
              {player.PLAYER_NAME}
            </div>
          );
        }
      },
      {
        accessorKey: 'POSITIONS',
        header: 'Pos',
      },
      {
        accessorKey: 'RANK',
        header: 'Rank',
      },
      {
        accessorKey: 'TOTAL_VALUE',
        header: 'Value',
      }
    ];

    // Add stat columns if available
    if (availableStats && availableStats.length) {
      const statColumns = availableStats
        .filter(stat => visibleColumns.includes(stat))
        .map(stat => ({
          accessorKey: stat,
          header: stat,
        }));

      return [...baseColumns, ...statColumns];
    }

    return baseColumns;
  }, [availableStats, visibleColumns, getPlayerDraftStatus]);

  // Set up the table
  const table = useReactTable({
    data: players || [],
    columns,
    state: {
      pagination,
    },
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  // Handle row click
  const handleRowClick = (player) => {
    setSelectedPlayer(player);
  };
  
  // Handle draft for me
  const handleDraftForMe = () => {
    if (selectedPlayer) {
      draftPlayerForMe(selectedPlayer);
      setSelectedPlayer(null);
    }
  };

  // Handle draft for others
  const handleDraftForOthers = () => {
    if (selectedPlayer) {
      draftPlayerForOthers(selectedPlayer);
      setSelectedPlayer(null);
    }
  };

  // Toggle column visibility
  const toggleColumn = (column) => {
    setVisibleColumns(prev => {
      if (prev.includes(column)) {
        return prev.filter(col => col !== column);
      } else {
        return [...prev, column];
      }
    });
  };

  return (
    <div className="player-table-container">
      <div className="table-controls">
        <button
          className="column-selector-toggle"
          onClick={() => setShowColumnSelector(!showColumnSelector)}
        >
          {showColumnSelector ? 'Hide Columns' : 'Show Columns'}
        </button>

        {showColumnSelector && (
          <div className="column-selector">
            <h4>Select Columns</h4>
            <div className="column-options">
              {availableStats && availableStats.map(stat => (
                <label key={stat} className="column-option">
                  <input
                    type="checkbox"
                    checked={visibleColumns.includes(stat)}
                    onChange={() => toggleColumn(stat)}
                  />
                  {stat}
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="table-wrapper">
        <table className="player-table">
          <thead>
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={
                      header.column.getIsSorted()
                        ? header.column.getIsSorted() === 'desc'
                          ? 'sort-desc'
                          : 'sort-asc'
                        : ''
                    }
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                    <span>
                      {header.column.getIsSorted()
                        ? header.column.getIsSorted() === 'desc'
                          ? ' 🔽'
                          : ' 🔼'
                        : ''}
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map(row => {
              const player = row.original;
              const isSelected = selectedPlayer && selectedPlayer.PLAYER_ID === player.PLAYER_ID;
              const status = getPlayerDraftStatus(player.PLAYER_ID);
              
              return (
                <tr
                  key={row.id}
                  onClick={() => handleRowClick(player)}
                  className={`
                    ${isSelected ? 'selected' : ''}
                    ${status === DRAFT_STATUS.DRAFTED_BY_YOU ? 'drafted-by-you' : ''}
                    ${status === DRAFT_STATUS.DRAFTED_BY_OTHERS ? 'drafted-by-others' : ''}
                  `}
                >
                  {row.getVisibleCells().map(cell => (
                    <td key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button
          onClick={() => table.setPageIndex(0)}
          disabled={!table.getCanPreviousPage()}
        >
          {'<<'}
        </button>{' '}
        <button
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          {'<'}
        </button>{' '}
        <span>
          Page{' '}
          <strong>
            {table.getState().pagination.pageIndex + 1} of{' '}
            {table.getPageCount()}
          </strong>{' '}
        </span>
        <button
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          {'>'}
        </button>{' '}
        <button
          onClick={() => table.setPageIndex(table.getPageCount() - 1)}
          disabled={!table.getCanNextPage()}
        >
          {'>>'}
        </button>{' '}
        <select
          value={table.getState().pagination.pageSize}
          onChange={e => {
            table.setPageSize(Number(e.target.value));
          }}
        >
          {[10, 25, 50, 100].map(pageSize => (
            <option key={pageSize} value={pageSize}>
              Show {pageSize}
            </option>
          ))}
        </select>
      </div>

      {selectedPlayer && (
        <div className="player-actions">
          <h3>Draft {selectedPlayer.PLAYER_NAME}</h3>
          <div className="action-buttons">
            <button
              className="draft-for-me"
              onClick={handleDraftForMe}
            >
              Draft for My Team
            </button>
            <button
              className="draft-for-others"
              onClick={handleDraftForOthers}
            >
              Drafted by Another Team
            </button>
            <button
              className="cancel"
              onClick={() => setSelectedPlayer(null)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlayerTable;
